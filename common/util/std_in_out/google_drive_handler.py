import io
import os
import re

import pandas as pd

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from framework.common.logger.message_type import MessageType


class GoogleDriveHandler:
    """
    Unica puerta de entrada y salida contra Google Drive / Google Sheets.

    La logica de negocio no sabe nada de Google: le pide a esta clase que le
    traiga las solapas como DataFrames y que escriba de vuelta los resultados.

    Autenticacion: service account. El JSON de la service account se pasa por
    parametro y la carpeta de Drive tiene que estar compartida con el mail de
    esa service account con permiso de edicion.
    """

    SCOPES = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets",
    ]

    GOOGLE_SHEET_MIME = "application/vnd.google-apps.spreadsheet"
    XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    def __init__(self, credentials_file, logger=None):

        if credentials_file is None or not os.path.exists(credentials_file):
            raise Exception(f"[GDRIVE] Credentials file not found: {credentials_file}")

        self.logger = logger
        self.credentials_file = credentials_file

        creds = Credentials.from_service_account_file(credentials_file, scopes=self.SCOPES)

        self.drive = build("drive", "v3", credentials=creds, cache_discovery=False)
        self.sheets = build("sheets", "v4", credentials=creds, cache_discovery=False)

    # ==================================================================
    # Logging
    # ==================================================================

    def _log(self, msg, msg_type=MessageType.INFO, job_id=None):
        if self.logger is not None:
            self.logger.do_log(msg, msg_type, job_id)
        else:
            print(msg)

    # ==================================================================
    # Resolucion de ids a partir de las URLs
    # ==================================================================

    @staticmethod
    def extract_folder_id(folder_url):
        """
        Acepta la URL completa de la carpeta compartida o directamente el id.
        """
        if folder_url is None:
            raise Exception("[GDRIVE] Empty folder url")

        match = re.search(r"/folders/([a-zA-Z0-9_\-]+)", folder_url)
        if match:
            return match.group(1)

        return folder_url.strip()

    @staticmethod
    def extract_file_id(file_url):
        """
        Acepta la URL completa de un archivo o directamente el id.
        """
        if file_url is None:
            raise Exception("[GDRIVE] Empty file url")

        match = re.search(r"/d/([a-zA-Z0-9_\-]+)", file_url)
        if match:
            return match.group(1)

        return file_url.strip()

    # ==================================================================
    # Busqueda de archivos dentro de la carpeta
    # ==================================================================

    def get_file(self, folder_id, file_name):
        """
        Busca un archivo por nombre exacto dentro de la carpeta.
        Devuelve el dict con id, name y mimeType.
        """
        safe_name = file_name.replace("'", "\\'")

        query = (
            f"'{folder_id}' in parents and "
            f"name = '{safe_name}' and "
            f"trashed = false"
        )

        response = self.drive.files().list(
            q=query,
            fields="files(id, name, mimeType)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()

        files = response.get("files", [])

        if len(files) == 0:
            raise Exception(f"[GDRIVE] File '{file_name}' not found in folder {folder_id}")

        if len(files) > 1:
            self._log(
                f"[GDRIVE] More than one file named '{file_name}'. Taking the first one.",
                MessageType.WARNING,
            )

        return files[0]

    def get_file_id(self, folder_id, file_name):
        return self.get_file(folder_id, file_name)["id"]

    def list_files(self, folder_id):
        response = self.drive.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="files(id, name, mimeType)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()

        return response.get("files", [])

    # ==================================================================
    # Lectura de solapas
    # ==================================================================

    def list_tabs(self, spreadsheet_id):
        meta = self.sheets.spreadsheets().get(
            spreadsheetId=spreadsheet_id,
            fields="sheets.properties.title",
        ).execute()

        return [s["properties"]["title"] for s in meta.get("sheets", [])]

    def read_tab(self, spreadsheet_id, tab_name):
        """
        Devuelve la solapa como DataFrame. La primera fila es el encabezado.
        Las filas mas cortas se completan con vacio para que no se rompa nada.
        """
        response = self.sheets.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=f"'{tab_name}'",
            valueRenderOption="UNFORMATTED_VALUE",
        ).execute()

        values = response.get("values", [])

        if len(values) == 0:
            return pd.DataFrame()

        header = [str(h).strip() for h in values[0]]
        width = len(header)

        rows = []
        for raw_row in values[1:]:
            row = list(raw_row)[:width]
            row = row + [""] * (width - len(row))
            rows.append(row)

        return pd.DataFrame(rows, columns=header)

    def read_all_tabs(self, spreadsheet_id, skip_tabs=None):
        skip = [t.upper() for t in (skip_tabs or [])]

        result = {}
        for tab in self.list_tabs(spreadsheet_id):
            if tab.upper() in skip:
                continue
            result[tab] = self.read_tab(spreadsheet_id, tab)

        return result

    # ==================================================================
    # Escritura de solapas
    # ==================================================================

    def write_tab(self, spreadsheet_id, tab_name, df, clear=True):
        """
        Escribe el DataFrame completo (encabezado + datos) arrancando en A1.
        """
        if clear:
            self.sheets.spreadsheets().values().clear(
                spreadsheetId=spreadsheet_id,
                range=f"'{tab_name}'",
                body={},
            ).execute()

        values = [list(df.columns)]
        for _, row in df.iterrows():
            values.append(["" if pd.isna(v) else v for v in row.tolist()])

        self.sheets.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=f"'{tab_name}'!A1",
            valueInputOption="USER_ENTERED",
            body={"values": values},
        ).execute()

        self._log(f"[GDRIVE] Tab '{tab_name}' written with {len(df)} rows")

    def write_rows(self, spreadsheet_id, tab_name, rows, start_cell="A1"):
        """
        Escribe filas sueltas desde una celda puntual, sin limpiar la solapa.
        Se usa para la solapa de control, que tiene encabezado y descripciones
        que no hay que pisar.
        """
        self.sheets.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=f"'{tab_name}'!{start_cell}",
            valueInputOption="USER_ENTERED",
            body={"values": rows},
        ).execute()

    def create_tab_if_missing(self, spreadsheet_id, tab_name):
        if tab_name in self.list_tabs(spreadsheet_id):
            return False

        self.sheets.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests": [{"addSheet": {"properties": {"title": tab_name}}}]},
        ).execute()

        self._log(f"[GDRIVE] Tab '{tab_name}' created")
        return True

    # ==================================================================
    # Bajada y subida del archivo entero
    # ==================================================================

    def download_file(self, file_id, local_path, mime_type=None):
        """
        Baja el archivo a disco. Si es una planilla de Google la exporta a xlsx,
        que es lo que sirve para guardarse una copia de respaldo de lo que se
        leyo antes de tocar nada.
        """
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)

        meta = self.drive.files().get(
            fileId=file_id,
            fields="mimeType, name",
            supportsAllDrives=True,
        ).execute()

        if meta["mimeType"] == self.GOOGLE_SHEET_MIME:
            request = self.drive.files().export_media(
                fileId=file_id,
                mimeType=mime_type or self.XLSX_MIME,
            )
        else:
            request = self.drive.files().get_media(fileId=file_id, supportsAllDrives=True)

        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()

        with open(local_path, "wb") as f:
            f.write(buffer.getvalue())

        self._log(f"[GDRIVE] File '{meta['name']}' downloaded to {local_path}")
        return local_path

    def upload_file(self, local_path, folder_id, file_name, mime_type=None):
        """
        Sube el archivo a la carpeta. Si ya existe uno con el mismo nombre le
        pisa el contenido en vez de crear un duplicado.
        """
        if not os.path.exists(local_path):
            raise Exception(f"[GDRIVE] Local file not found: {local_path}")

        media = MediaFileUpload(
            local_path,
            mimetype=mime_type or self.XLSX_MIME,
            resumable=True,
        )

        try:
            existing_id = self.get_file_id(folder_id, file_name)
        except Exception:
            existing_id = None

        if existing_id is not None:
            self.drive.files().update(
                fileId=existing_id,
                media_body=media,
                supportsAllDrives=True,
            ).execute()

            self._log(f"[GDRIVE] File '{file_name}' updated")
            return existing_id

        created = self.drive.files().create(
            body={"name": file_name, "parents": [folder_id]},
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        ).execute()

        self._log(f"[GDRIVE] File '{file_name}' created")
        return created["id"]
