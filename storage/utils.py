def get_db_url(user: str, password: str, host: str, port: int, dname: str):
    return f'postgresql+asyncpg://{user}:{password}@{host}:{port}/{dname}?target_session_attrs=read-write'
