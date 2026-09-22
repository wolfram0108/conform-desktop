"""Base of the conform models that travel over the API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ApiModel(BaseModel):
    """In the published schema of an answer a field with a default is always present, as it is on
    the wire: a client generated from the schema needs no guesses."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)
