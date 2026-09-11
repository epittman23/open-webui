from fastapi import APIRouter

from open_webui.routers.benchmarks import answers, compare, live, report, serve, tests

router = APIRouter()
router.include_router(serve.router, prefix='/serve')
router.include_router(live.router, prefix='/live')
router.include_router(tests.router, prefix='/tests')
router.include_router(compare.router, prefix='/compare')
router.include_router(answers.router, prefix='/answers')
router.include_router(report.router, prefix='/report')

try:
    from open_webui.routers.benchmarks import tune

    router.include_router(tune.router, prefix='/tune')
except ImportError:
    # benchmarks.tune (the config-search engine) is ported separately and
    # may not exist yet in every checkout of this branch -- every other
    # page works without it, so its absence should not break the rest of
    # the Benchmarks feature.
    pass
