# # database_conn.py

# import asyncpg
# from typing import AsyncGenerator

# # [주의] 이 정보는 실제 서비스에서는 환경 변수로 관리해야 합니다.
# DB_USER = "neondb_owner"
# DB_PASSWORD = "npg_rFA9DgloybO0"  # 실제 비밀번호 사용
# DB_HOST = "ep-rapid-truth-ad2e6rnq-pooler.c-2.us-east-1.aws.neon.tech"
# DB_NAME = "neondb"
# DB_PORT = 5432
# DB_SSL = 'require'

# # 글로벌 DB 연결 풀 변수
# pool: asyncpg.Pool = None

# # 서버 시작 시 호출 (FastAPI @app.on_event("startup")에서 사용)
# async def connect_to_db():
#     global pool
#     print("DB 연결 풀 초기화 중...")
#     try:
#         pool = await asyncpg.create_pool(
#             user=DB_USER,
#             password=DB_PASSWORD,
#             host=DB_HOST,
#             database=DB_NAME,
#             port=DB_PORT,
#             ssl=DB_SSL 
#         )
#         print("DB 연결 성공.")
#     except Exception as e:
#         print(f"DB 연결 실패: {e}")
#         raise e

# # 서버 종료 시 호출 (FastAPI @app.on_event("shutdown")에서 사용)
# async def close_db_connection():
#     global pool
#     if pool:
#         print("DB 연결 풀 종료 중...")
#         await pool.close()
#         print("DB 연결 풀 종료 완료.")

# # API 엔드포인트에 DB 연결 객체를 주입하는 함수 (FastAPI Depends에서 사용)
# async def get_db_connection() -> AsyncGenerator[asyncpg.Connection, None]:
#     """DB 연결 풀에서 커넥션을 가져와 엔드포인트에 제공합니다."""
#     global pool
#     if not pool:
#         raise ConnectionError("DB 연결 풀이 초기화되지 않았습니다.")

#     # async with 구문으로 연결을 가져와 yield 후, 자동으로 풀에 반환합니다.
#     async with pool.acquire() as connection:
#         yield connection