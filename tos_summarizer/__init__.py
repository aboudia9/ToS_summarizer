# The pipeline (and torch) is only imported when actually used, so lightweight parts
# like extract.py can be imported on their own (the web app parses uploads in a separate
# process that never needs the models)
__all__ = ["process_tos_document", "print_results", "ClauseResult"]


def __getattr__(name):
    if name in __all__:
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(name)
