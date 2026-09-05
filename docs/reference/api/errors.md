# Exceptions

Every error polspec raises descends from `PolspecError`, so one `except`
clause catches the lot. Most also descend from the built-in they replaced, so
existing `except ValueError` handlers keep working. The
[Errors and findings](../errors.md) page explains when each is raised.

## PolspecError

::: polspec.PolspecError

## SpecError

::: polspec.SpecError

## GenerationError

::: polspec.GenerationError

## ValidationError

::: polspec.ValidationError

## SerializationError

::: polspec.SerializationError

## RegistryError

::: polspec.RegistryError
