-- =============================================================================
-- Migration: Create all tables required by the Job Matching AI Service
-- Run this ONCE against your SQL Server database before starting the service.
-- The application also auto-creates these tables on startup if they don't exist.
-- =============================================================================

-- ─── 1. CV Parsed Data ───────────────────────────────────────────────────────
-- One row per user (UNIQUE on application_user_id).
-- On re-upload the old row is deleted and a fresh row is inserted.

IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'cv_parsed_data'
)
BEGIN
    CREATE TABLE cv_parsed_data (
        id                  INT IDENTITY(1,1) PRIMARY KEY,
        application_user_id NVARCHAR(255) NOT NULL UNIQUE,
        full_name           NVARCHAR(512),
        email               NVARCHAR(512),
        phone               NVARCHAR(128),
        location            NVARCHAR(512),
        languages           NVARCHAR(MAX),
        skills              NVARCHAR(MAX),
        certifications      NVARCHAR(MAX),
        education           NVARCHAR(MAX),
        experience          NVARCHAR(MAX),
        summary             NVARCHAR(MAX),
        raw_json            NVARCHAR(MAX),
        created_at          DATETIME DEFAULT GETDATE(),
        updated_at          DATETIME DEFAULT GETDATE()
    );
    PRINT 'Created table: cv_parsed_data';
END
ELSE
    PRINT 'Table already exists: cv_parsed_data';
GO

-- ─── 2. Prompt Roadmaps ───────────────────────────────────────────────────────
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'prompt_roadmaps'
)
BEGIN
    CREATE TABLE prompt_roadmaps (
        id          INT IDENTITY(1,1) PRIMARY KEY,
        user_id     NVARCHAR(255) NOT NULL,
        title       NVARCHAR(512),
        description NVARCHAR(MAX),
        created_at  DATETIME DEFAULT GETDATE()
    );
    CREATE INDEX IX_prompt_roadmaps_user_id ON prompt_roadmaps(user_id);
    PRINT 'Created table: prompt_roadmaps';
END
ELSE
    PRINT 'Table already exists: prompt_roadmaps';
GO

-- ─── 3. Interview Questions ───────────────────────────────────────────────────
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'interview_questions'
)
BEGIN
    CREATE TABLE interview_questions (
        id              INT IDENTITY(1,1) PRIMARY KEY,
        job_id          NVARCHAR(255) NOT NULL,
        job_title       NVARCHAR(512),
        questions_json  NVARCHAR(MAX),
        created_at      DATETIME DEFAULT GETDATE()
    );
    CREATE INDEX IX_interview_questions_job_id ON interview_questions(job_id);
    PRINT 'Created table: interview_questions';
END
ELSE
    PRINT 'Table already exists: interview_questions';
GO

PRINT '=== Migration complete ===';
