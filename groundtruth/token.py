import argparse
from datetime import datetime, timedelta, timezone

import jwt

from groundtruth.config import settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", required=True)
    parser.add_argument("--groups", nargs="*", default=[])
    args = parser.parse_args()
    if len(settings.jwt_secret) < 32:
        raise SystemExit("Set JWT_SECRET to at least 32 random characters")
    print(jwt.encode({"sub": args.sub, "groups": args.groups, "exp": datetime.now(timezone.utc) + timedelta(hours=8)}, settings.jwt_secret, algorithm="HS256"))


if __name__ == "__main__":
    main()
