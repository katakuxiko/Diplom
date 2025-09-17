To use LLM need LM STUDIO

DB PostgresSQL

if you have own config, use command in bash like
```
 export SERVER_ADDR=":8080"
```

swagger initialization:

```
swag init -g cmd/server/main.go -d ./ -o ./docs
```