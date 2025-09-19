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
swag init -g cmd/server/main.go -d ./ -o ./docs
```