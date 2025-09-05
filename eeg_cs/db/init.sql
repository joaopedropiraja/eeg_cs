PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS evaluations (
	id               INTEGER PRIMARY KEY AUTOINCREMENT,
	dataset          TEXT    NOT NULL,
	channel          TEXT    NOT NULL,
	sparsifying_matrix TEXT  NOT NULL,
	algorithm        TEXT    NOT NULL,
	prd              REAL    NOT NULL,
	nmse             REAL    NOT NULL,
	sndr             REAL    NOT NULL,
	time_s           REAL    NOT NULL,
	compression_rate INTEGER NOT NULL,
	segment_length   INTEGER NOT NULL,
	created_at       DATETIME DEFAULT CURRENT_TIMESTAMP
);


CREATE INDEX IF NOT EXISTS idx_eval_dataset ON evaluations(dataset);
CREATE INDEX IF NOT EXISTS idx_eval_dataset_channel ON evaluations(dataset, channel);
CREATE INDEX IF NOT EXISTS idx_eval_basis_algo ON evaluations(sparsifying_matrix, algorithm);
CREATE INDEX IF NOT EXISTS idx_eval_cr_seglen ON evaluations(compression_rate, segment_length);

