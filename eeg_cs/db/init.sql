PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS evaluations (
	id               	 INTEGER PRIMARY KEY AUTOINCREMENT,
	dataset            TEXT    NOT NULL,
	fs                 INTEGER NOT NULL,
	channel            TEXT    NOT NULL,
	file_name          TEXT    NOT NULL,
	start_time_idx     INTEGER NOT NULL,
	sensing_matrix     TEXT    NOT NULL,
	sparsifying_matrix TEXT    NOT NULL,
	algorithm          TEXT    NOT NULL,
	mean_freq          REAL    NOT NULL,
	prd                REAL    NOT NULL,
	nmse               REAL    NOT NULL,
	sndr               REAL    NOT NULL,
	ssim               REAL    NOT NULL,
	elapsed_time_s     REAL    NOT NULL,
	compression_rate   INTEGER NOT NULL,
	segment_length_s   INTEGER NOT NULL,
	created_at         DATETIME DEFAULT CURRENT_TIMESTAMP
);

