import asyncio
from item_predictors import MfuPredictor

from aiohttp import web


@web.middleware
async def cors_middleware(request, handler):
    response = await handler(request)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

async def run_app():
    mfu_web = MfuPredictor()
    await mfu_web.setup()

    app = web.Application(middlewares=[cors_middleware])
    app.router.add_post('/getCluster', mfu_web.get_cluster)


    await web._run_app(app)

if __name__ == "__main__":
    asyncio.run(run_app())
