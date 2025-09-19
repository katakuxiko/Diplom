package storage

import (
	"bytes"
	"context"
	"mime/multipart"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

type MinioStorage struct {
	client *minio.Client
	bucket string
}

func NewMinioStorage(endpoint, accessKey, secretKey, bucket string, useSSL bool) (*MinioStorage, error) {
	client, err := minio.New(endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(accessKey, secretKey, ""),
		Secure: useSSL,
	})
	if err != nil {
		return nil, err
	}

	ctx := context.Background()
	exists, err := client.BucketExists(ctx, bucket)
	if err != nil {
		return nil, err
	}
	if !exists {
		if err := client.MakeBucket(ctx, bucket, minio.MakeBucketOptions{}); err != nil {
			return nil, err
		}
	}

	return &MinioStorage{client: client, bucket: bucket}, nil
}

// UploadFile сохраняет файл в MinIO и возвращает objectName
func (s *MinioStorage) UploadFile(objectName string, file multipart.File, fileHeader *multipart.FileHeader) (string, error) {
	buf := new(bytes.Buffer)
	if _, err := buf.ReadFrom(file); err != nil {
		return "", err
	}

	_, err := s.client.PutObject(
		context.Background(),
		s.bucket,
		objectName,
		bytes.NewReader(buf.Bytes()),
		int64(buf.Len()),
		minio.PutObjectOptions{
			ContentType: fileHeader.Header.Get("Content-Type"),
		},
	)
	if err != nil {
		return "", err
	}

	return objectName, nil
}

// DownloadFile возвращает ссылку на файл (presigned URL)
func (s *MinioStorage) DownloadFile(objectName string) (string, error) {
	url, err := s.client.PresignedGetObject(context.Background(), s.bucket, objectName, 0, nil)
	if err != nil {
		return "", err
	}
	return url.String(), nil
}

// DeleteFile удаляет файл из MinIO
func (s *MinioStorage) DeleteFile(objectName string) error {
	return s.client.RemoveObject(context.Background(), s.bucket, objectName, minio.RemoveObjectOptions{})
}
