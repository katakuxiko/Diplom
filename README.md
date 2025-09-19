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
```

minio download https://www.min.io/download?platform=windows

minio start
```
 .\minio.exe server C:\minio\data --console-address ":9090"
```