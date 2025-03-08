-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create articles table
CREATE TABLE IF NOT EXISTS articles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    published_date TIMESTAMP,
    author TEXT,
    source_domain TEXT NOT NULL,
    category TEXT,
    keywords TEXT[],
    vector_embedding VECTOR(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create sources table
CREATE TABLE IF NOT EXISTS sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    domain TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    base_url TEXT NOT NULL,
    scraper_type TEXT NOT NULL,
    active BOOLEAN DEFAULT TRUE,
    last_crawled TIMESTAMP WITH TIME ZONE,
    crawl_frequency INTEGER DEFAULT 24, -- in hours
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create crawl_logs table
CREATE TABLE IF NOT EXISTS crawl_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID REFERENCES sources(id),
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    articles_found INTEGER DEFAULT 0,
    articles_added INTEGER DEFAULT 0,
    articles_updated INTEGER DEFAULT 0,
    status TEXT DEFAULT 'in_progress',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_articles_source_domain ON articles(source_domain);
CREATE INDEX IF NOT EXISTS idx_articles_published_date ON articles(published_date);
CREATE INDEX IF NOT EXISTS idx_sources_domain ON sources(domain);
CREATE INDEX IF NOT EXISTS idx_crawl_logs_source_id ON crawl_logs(source_id);
CREATE INDEX IF NOT EXISTS idx_crawl_logs_status ON crawl_logs(status);

-- Create vector index for semantic search
CREATE INDEX IF NOT EXISTS idx_articles_vector_embedding ON articles USING ivfflat (vector_embedding vector_cosine_ops) WITH (lists = 100);

-- Insert initial sources from news-sources.txt
INSERT INTO sources (domain, name, base_url, scraper_type)
VALUES 
    ('markettimes.vn', 'Market Times', 'https://markettimes.vn/', 'newspaper'),
    ('intelligenceonline.com', 'Intelligence Online', 'https://www.intelligenceonline.com/', 'newspaper'),
    ('vietnamnet.vn', 'VietnamNet', 'https://vietnamnet.vn/kinh-doanh', 'newspaper'),
    ('vietnamexport.com', 'Vietnam Export', 'https://vietnamexport.com/', 'newspaper'),
    ('businessinsider.com', 'Business Insider', 'https://www.businessinsider.com/', 'newspaper'),
    ('wsj.com', 'Wall Street Journal', 'https://www.wsj.com/', 'newspaper'),
    ('nytimes.com', 'New York Times', 'https://www.nytimes.com/', 'newspaper'),
    ('theguardian.com', 'The Guardian', 'https://www.theguardian.com/us', 'newspaper'),
    ('bloomberg.com', 'Bloomberg', 'https://www.bloomberg.com/', 'newspaper')
ON CONFLICT (domain) DO NOTHING; 