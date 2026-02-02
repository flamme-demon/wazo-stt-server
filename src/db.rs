use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;
use std::sync::Mutex;

pub struct Database {
    conn: Mutex<Connection>,
}

#[derive(Debug, Clone)]
pub struct TranscriptionRecord {
    pub id: String,
    pub user_uuid: String,
    pub message_id: String,
    pub status: String,
    pub text: Option<String>,
    pub duration: Option<f64>,
    pub error: Option<String>,
    pub created_at: i64,
    pub completed_at: Option<i64>,
}

impl Database {
    pub fn new(path: &Path) -> Result<Self> {
        let conn = Connection::open(path).context("Failed to open database")?;

        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS transcriptions (
                id TEXT PRIMARY KEY,
                user_uuid TEXT NOT NULL,
                message_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                text TEXT,
                duration REAL,
                error TEXT,
                created_at INTEGER NOT NULL,
                completed_at INTEGER
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_user_message
            ON transcriptions(user_uuid, message_id);

            CREATE INDEX IF NOT EXISTS idx_status
            ON transcriptions(status);

            CREATE INDEX IF NOT EXISTS idx_created_at
            ON transcriptions(created_at);
            ",
        )
        .context("Failed to create tables")?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    pub fn find_by_user_and_message(
        &self,
        user_uuid: &str,
        message_id: &str,
    ) -> Result<Option<TranscriptionRecord>> {
        let conn = self.conn.lock().unwrap();

        let result = conn
            .query_row(
                "SELECT id, user_uuid, message_id, status, text, duration, error, created_at, completed_at
                 FROM transcriptions
                 WHERE user_uuid = ?1 AND message_id = ?2",
                params![user_uuid, message_id],
                |row| {
                    Ok(TranscriptionRecord {
                        id: row.get(0)?,
                        user_uuid: row.get(1)?,
                        message_id: row.get(2)?,
                        status: row.get(3)?,
                        text: row.get(4)?,
                        duration: row.get(5)?,
                        error: row.get(6)?,
                        created_at: row.get(7)?,
                        completed_at: row.get(8)?,
                    })
                },
            )
            .optional()
            .context("Failed to query transcription")?;

        Ok(result)
    }

    pub fn find_by_id(&self, id: &str) -> Result<Option<TranscriptionRecord>> {
        let conn = self.conn.lock().unwrap();

        let result = conn
            .query_row(
                "SELECT id, user_uuid, message_id, status, text, duration, error, created_at, completed_at
                 FROM transcriptions
                 WHERE id = ?1",
                params![id],
                |row| {
                    Ok(TranscriptionRecord {
                        id: row.get(0)?,
                        user_uuid: row.get(1)?,
                        message_id: row.get(2)?,
                        status: row.get(3)?,
                        text: row.get(4)?,
                        duration: row.get(5)?,
                        error: row.get(6)?,
                        created_at: row.get(7)?,
                        completed_at: row.get(8)?,
                    })
                },
            )
            .optional()
            .context("Failed to query transcription")?;

        Ok(result)
    }

    pub fn insert(&self, record: &TranscriptionRecord) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "INSERT INTO transcriptions (id, user_uuid, message_id, status, text, duration, error, created_at, completed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                record.id,
                record.user_uuid,
                record.message_id,
                record.status,
                record.text,
                record.duration,
                record.error,
                record.created_at,
                record.completed_at,
            ],
        )
        .context("Failed to insert transcription")?;

        Ok(())
    }

    pub fn update_status(&self, id: &str, status: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "UPDATE transcriptions SET status = ?2 WHERE id = ?1",
            params![id, status],
        )
        .context("Failed to update status")?;

        Ok(())
    }

    pub fn update_completed(
        &self,
        id: &str,
        text: &str,
        duration: f64,
        completed_at: i64,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "UPDATE transcriptions
             SET status = 'completed', text = ?2, duration = ?3, completed_at = ?4
             WHERE id = ?1",
            params![id, text, duration, completed_at],
        )
        .context("Failed to update completed transcription")?;

        Ok(())
    }

    pub fn update_failed(&self, id: &str, error: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "UPDATE transcriptions SET status = 'failed', error = ?2 WHERE id = ?1",
            params![id, error],
        )
        .context("Failed to update failed transcription")?;

        Ok(())
    }

    pub fn cleanup_old(&self, max_age_seconds: i64) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let cutoff = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
            - max_age_seconds;

        let deleted = conn
            .execute(
                "DELETE FROM transcriptions WHERE created_at < ?1",
                params![cutoff],
            )
            .context("Failed to cleanup old transcriptions")?;

        Ok(deleted)
    }

    pub fn get_stats(&self) -> Result<DbStats> {
        let conn = self.conn.lock().unwrap();

        let total: i64 = conn.query_row(
            "SELECT COUNT(*) FROM transcriptions",
            [],
            |row| row.get(0),
        )?;

        let completed: i64 = conn.query_row(
            "SELECT COUNT(*) FROM transcriptions WHERE status = 'completed'",
            [],
            |row| row.get(0),
        )?;

        let failed: i64 = conn.query_row(
            "SELECT COUNT(*) FROM transcriptions WHERE status = 'failed'",
            [],
            |row| row.get(0),
        )?;

        let pending: i64 = conn.query_row(
            "SELECT COUNT(*) FROM transcriptions WHERE status IN ('queued', 'processing')",
            [],
            |row| row.get(0),
        )?;

        Ok(DbStats {
            total: total as usize,
            completed: completed as usize,
            failed: failed as usize,
            pending: pending as usize,
        })
    }
}

#[derive(Debug, Clone)]
pub struct DbStats {
    pub total: usize,
    pub completed: usize,
    pub failed: usize,
    pub pending: usize,
}
