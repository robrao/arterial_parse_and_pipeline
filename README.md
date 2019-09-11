# arterial_parse_and_pipeline
Parse contour files and put data into TFRecord for training.

Expects that final_data.tar.gz is extracted with in this repo, so we have final_data/link.csv, final_data/contour_files, and final_data/dicoms. User
will also have to `pip install -r requirements.txt` to have all needed modules.
