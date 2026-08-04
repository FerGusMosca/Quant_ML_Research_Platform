# Snippets — lo que hay que pegar a mano

Son 6 pegadas sobre archivos tuyos. Todo lo demás son archivos nuevos que se
copian y listo.

---

## 1) `configs/commands_mgr.ini` — agregar en `[SETTINGS]`

```ini
SEC_USER_AGENT=Seeking Bias Research alien.zimzum@gmail.com
```

La SEC rechaza los requests que no traen un User-Agent identificatorio con
nombre y mail. Sin esto el job devuelve 403 en todas.

---

## 2) `common/util/std_in_out/ml_settings_loader.py`

Junto a las otras líneas de `[SETTINGS]`:

```python
            config_settings["SEC_USER_AGENT"] = config['SETTINGS']['SEC_USER_AGENT']
```

---

## 3) `controllers/main_dashboard_controller.py`

Con los otros imports de controllers:

```python
from controllers.sec_securities_controller import SECSecuritiesController
```

Y al lado del registro de `chunk_mgmt_ctrl`:

```python
        self.sec_securities_ctrl = SECSecuritiesController(config_settings, logger)
        self.app.include_router(self.sec_securities_ctrl.router, prefix="/sec_securities")
```

---

## 4) `templates/base.html` — item en el sidebar

Debajo del item de Chunk Management, dentro del bloque `Data`:

```html
    <a href="/sec_securities" class="nav-item {% if active_page == 'sec_securities' %}active{% endif %}">
      <span class="nav-icon">🏷️</span> SEC Securities
    </a>
```

---

## 5) `bias_mgmt_console.py`

### 5.a — en `show_commands()`, bajo Financial Reports

```python
    print("#71-DownloadSECSecuritiesMetadata [top*] [retry_errors*]")
    print("#72-DownloadSECSecurityMetadata [symbol]")
    print("#73-TagSecuritiesFromCSV [tag] [csv_path] [tag_group*]")
```

### 5.b — handlers de comando (al lado de `process_download_sec_securities`)

```python
def process_download_sec_securities_metadata(cmd):
    top = ParamReader.get_param(cmd, "top", True, None)
    retry_errors = ParamReader.get_param(cmd, "retry_errors", True, False)
    process_download_sec_securities_metadata_logic(top, retry_errors)


def process_download_sec_security_metadata(cmd):
    symbol = ParamReader.get_param(cmd, "symbol")
    process_download_sec_security_metadata_logic(symbol)


def process_tag_securities_from_csv(cmd):
    tag = ParamReader.get_param(cmd, "tag")
    csv_path = ParamReader.get_param(cmd, "csv_path")
    tag_group = ParamReader.get_param(cmd, "tag_group", True, "CUSTOM")
    process_tag_securities_from_csv_logic(tag, csv_path, tag_group)
```

### 5.c — entrypoints de lógica (al lado de `process_download_sec_securities_logic`)

```python
def __build_sec_metadata_orchestation__(logger):
    loader = MLSettingsLoader()
    config_settings = loader.load_settings("./configs/commands_mgr.ini")
    return SECMetadataOrchestationLogic(
        config_settings["ml_reports_conn_str"],
        logger,
        config_settings["SEC_USER_AGENT"]
    )


def process_download_sec_securities_metadata_logic(top=None, retry_errors=False):
    logger = Logger()
    try:
        logger.print("[SEC-META] Starting SEC securities metadata download", MessageType.INFO)
        orchestation = __build_sec_metadata_orchestation__(logger)
        result = orchestation.process_download_all_metadata(
            top=int(top) if top else None,
            include_errors=str(retry_errors).lower() in ("true", "1", "yes")
        )
        logger.print(f"[SEC-META] ✅ ok={result['ok']} fail={result['failed']} "
                     f"de {result['total']}", MessageType.INFO)
    except Exception as e:
        print(traceback.format_exc())
        logger.print(f"[SEC-META] ❌ Critical error: {str(e)}", MessageType.ERROR)


def process_download_sec_security_metadata_logic(symbol):
    logger = Logger()
    try:
        orchestation = __build_sec_metadata_orchestation__(logger)
        result = orchestation.process_download_single_metadata(symbol=symbol)
        if result.get("ok"):
            logger.print(f"[SEC-META] ✅ {symbol} actualizada", MessageType.INFO)
        else:
            logger.print(f"[SEC-META] ❌ {symbol} no se pudo actualizar", MessageType.ERROR)
    except Exception as e:
        print(traceback.format_exc())
        logger.print(f"[SEC-META] ❌ Critical error on {symbol}: {str(e)}", MessageType.ERROR)


def process_tag_securities_from_csv_logic(tag, csv_path, tag_group="CUSTOM"):
    logger = Logger()
    try:
        orchestation = __build_sec_metadata_orchestation__(logger)
        result = orchestation.process_tag_securities_from_csv(tag, csv_path,
                                                              tag_group=tag_group)
        logger.print(f"[SEC-TAG] ✅ {result['tag_code']}: {result['tagged']} taggeadas "
                     f"de {result['read']} leídas, {len(result['not_found'])} sin match",
                     MessageType.INFO)
        if result["not_found"]:
            logger.print(f"[SEC-TAG] sin match: {', '.join(result['not_found'][:40])}",
                         MessageType.WARNING)
    except Exception as e:
        print(traceback.format_exc())
        logger.print(f"[SEC-TAG] ❌ Critical error: {str(e)}", MessageType.ERROR)
```

### 5.d — import arriba del archivo

```python
from logic_layer.sec_metadata_orchestation_logic import SECMetadataOrchestationLogic
```

### 5.e — en `process_commands`, al lado de `DownloadSECSecurities`

```python
    elif cmd_param_list[0] == "DownloadSECSecuritiesMetadata":
        process_download_sec_securities_metadata(cmd)
    elif cmd_param_list[0] == "DownloadSECSecurityMetadata":
        process_download_sec_security_metadata(cmd)
    elif cmd_param_list[0] == "TagSecuritiesFromCSV":
        process_tag_securities_from_csv(cmd)
```

---

## 6) Opcional — `common/enums/report_type.py` + `process_run_report`

Si querés dispararlo también desde el menú de reportes:

```python
    DOWNLOAD_SEC_METADATA = "download_sec_metadata"
    DOWNLOAD_SEC_METADATA_SINGLE_SECURITY = "download_sec_metadata_single_security"
```

Y en `process_run_report`, antes del `else`:

```python
        elif report_key.lower() == ReportType.DOWNLOAD_SEC_METADATA.value:
            self._run_download_sec_metadata(job_id)
        elif report_key.lower() == ReportType.DOWNLOAD_SEC_METADATA_SINGLE_SECURITY.value:
            self._run_download_sec_metadata_single_security(symbol, job_id)
```

Con estos dos métodos en la misma clase:

```python
    def _run_download_sec_metadata(self, job_id=None):
        from logic_layer.sec_metadata_orchestation_logic import SECMetadataOrchestationLogic
        orchestation = SECMetadataOrchestationLogic(self.ml_reports_conn_str, self.logger,
                                                    self.sec_user_agent)
        return orchestation.process_download_all_metadata()

    def _run_download_sec_metadata_single_security(self, symbol, job_id=None):
        from logic_layer.sec_metadata_orchestation_logic import SECMetadataOrchestationLogic
        orchestation = SECMetadataOrchestationLogic(self.ml_reports_conn_str, self.logger,
                                                    self.sec_user_agent)
        return orchestation.process_download_single_metadata(symbol=symbol)
```

`ReportsOrchestationLogic.__init__` tiene que guardarse `self.ml_reports_conn_str` y
`self.sec_user_agent` para que esto funcione — hoy no los guarda.
