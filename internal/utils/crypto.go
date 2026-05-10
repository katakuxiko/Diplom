package utils

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/base64"
	"io"
	"os"
)

// getKey возвращает 32-байтовый ключ из переменной окружения SETTINGS_CRYPTO_KEY
func getKey() []byte {
    k := os.Getenv("SETTINGS_CRYPTO_KEY")
    if len(k) >= 32 {
        return []byte(k)[:32]
    }
    // если ключ короче — дополняем нулями (не идеально для продакшна)
    b := make([]byte, 32)
    copy(b, []byte(k))
    return b
}

// EncryptString шифрует строку AES-GCM и возвращает base64
func EncryptString(plain string) (string, error) {
    key := getKey()
    if len(key) == 0 {
        return plain, nil
    }
    block, err := aes.NewCipher(key)
    if err != nil {
        return "", err
    }
    aesgcm, err := cipher.NewGCM(block)
    if err != nil {
        return "", err
    }
    nonce := make([]byte, aesgcm.NonceSize())
    if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
        return "", err
    }
    ct := aesgcm.Seal(nonce, nonce, []byte(plain), nil)
    return base64.StdEncoding.EncodeToString(ct), nil
}

// DecryptString дешифрует base64 AES-GCM
func DecryptString(encoded string) (string, error) {
    key := getKey()
    if len(key) == 0 {
        return encoded, nil
    }
    data, err := base64.StdEncoding.DecodeString(encoded)
    if err != nil {
        return "", err
    }
    block, err := aes.NewCipher(key)
    if err != nil {
        return "", err
    }
    aesgcm, err := cipher.NewGCM(block)
    if err != nil {
        return "", err
    }
    nonceSize := aesgcm.NonceSize()
    if len(data) < nonceSize {
        return "", err
    }
    nonce, ct := data[:nonceSize], data[nonceSize:]
    pt, err := aesgcm.Open(nil, nonce, ct, nil)
    if err != nil {
        return "", err
    }
    return string(pt), nil
}
