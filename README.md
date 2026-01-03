To use LLM need LM STUDIO

DB PostgresSQL

if you have own config, use .env file
```
 SERVER_ADDR=":8080"
```

Start with refresh

```
go install github.com/air-verse/air@latest
```
```
air
```

swagger initialization:

```
swag init -g main.go -d ./ -o ./docs
~/go/bin/swag init -g main.go -d ./ -o ./docs
```

minio download https://www.min.io/download?platform=windows

minio start
```
 .\minio.exe server C:\minio\data --console-address ":9090"

 /opt/homebrew/opt/minio/bin/minio server --certs-dir\=/opt/homebrew/etc/minio/certs --address\=:9000 /opt/homebrew/var/minio

 docker compose up -d postgres minio minio-client

```

To extract text need install 
```
https://github.com/oschwartz10612/poppler-windows/releases
```

pg vector

```
https://github.com/pgvector/pgvector
```

docker compose up -d postgres minio minio-client