# Exceptions

Every error polspec raises descends from `PolspecError`, so one `except`
clause catches the lot, and none is also a built-in error: `except
ValueError` catches argument misuse, not a spec's complaint. *Since 2.0.* The
[Errors and findings](../errors.md) page explains when each is raised.

## PolspecError

::: polspec.PolspecError

## SpecError

::: polspec.SpecError

## GenerationError

::: polspec.GenerationError

## ValidationError

::: polspec.ValidationError

## MultiValidationError

::: polspec.MultiValidationError

## SerializationError

::: polspec.SerializationError

## RegistryError

::: polspec.RegistryError
