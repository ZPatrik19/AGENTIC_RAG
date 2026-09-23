"""Source allowlist and validated manifest; runtime statuses are not committed."""
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field, model_validator

DOMAINS = ('vehicle', 'employment')
ALLOWED_HOSTS = frozenset({'dap.gov.hu', 'nav.gov.hu', 'nfsz.munka.hu', 'njt.hu', 'njt.jog.gov.hu', 'neak.gov.hu', 'www.neak.gov.hu', 'magyarorszag.hu', 'www.magyarorszag.hu', 'mabisz.hu'})


class Source(BaseModel):
    id: str = Field(pattern=r'^[a-z0-9][a-z0-9-]{2,90}$')
    title: str = Field(min_length=4)
    url: str
    domain: Literal['vehicle', 'employment']
    type: Literal['html', 'pdf']
    destination: Literal['vehicle', 'employment']
    processing_status: Literal['pending', 'active', 'disabled'] = 'pending'
    required: bool = False
    role: Literal['buyer', 'seller', 'general'] = 'general'
    legal_sections: list[str] = Field(default_factory=list)
    source_page: str | None = None
    topics: list[str] = Field(default_factory=list)
    authority: str = 'official'
    priority: int = Field(default=50, ge=0, le=100)

    @model_validator(mode='after')
    def check_urls(self):
        for url in (self.url, self.source_page):
            if url is None:
                continue
            parts = urlparse(url)
            if parts.scheme != 'https' or parts.hostname not in ALLOWED_HOSTS:
                raise ValueError(f'URL not in HTTPS official allowlist: {url}')
            if parts.username or parts.password or parts.port:
                raise ValueError('Credentials and custom ports are not permitted')
        if self.destination != self.domain:
            raise ValueError('Destination must equal domain')
        return self


class Manifest(BaseModel):
    sources: list[Source] = Field(min_length=1)

    @model_validator(mode='after')
    def unique_sources(self):
        ids = [source.id for source in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate document ID in manifest')
        return self


def load_manifest(path: Path) -> Manifest:
    return Manifest.model_validate(yaml.safe_load(path.read_text(encoding='utf-8')))
