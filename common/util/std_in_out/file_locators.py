import os

from framework.common.logger.message_type import MessageType


class FileLocators ():


    @staticmethod
    def enumerate_all_files(file_folder,logger, filters=[],job_id=None):
        pass  # TODO enumerates all files in a folder
        if not os.path.isdir(file_folder):
            logger.do_log(
                f"[TAGGING] ⚠ Folder not found: {file_folder}",
                MessageType.WARNING,job_id
            )
            return []

        matched_files = []
        skipped_files = []

        try:
            all_files = [
                f for f in os.listdir(file_folder)
                if os.path.isfile(os.path.join(file_folder, f))
            ]
        except Exception as e:
            logger.do_log(
                f"[TAGGING] ❌ Failed listing files in {file_folder} → {e}",
                MessageType.ERROR,job_id
            )
            return []

        for fname in all_files:
            fname_l = fname.lower()

            if any(sym in fname_l for sym in filters):
                matched_files.append(os.path.join(file_folder, fname))
            else:
                skipped_files.append(fname)

        logger.do_log(
            f"[TAGGING] 📄 Files found={len(all_files)} | matched={len(matched_files)} | skipped={len(skipped_files)}",
            MessageType.INFO,job_id
        )

        if not matched_files:
            logger.do_log(
                f"[TAGGING] ⚠ No matching documents for folder {file_folder}",
                MessageType.WARNING,job_id
            )
            return []
        else:
            return  matched_files



