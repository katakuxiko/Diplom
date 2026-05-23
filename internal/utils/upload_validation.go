package utils

import (
	"fmt"
	"mime/multipart"
	"net/http"
	"path/filepath"
	"strings"
)

const MaxPDFSizeBytes int64 = 20 * 1024 * 1024 // 20 MB

func ValidatePDFUpload(fileHeader *multipart.FileHeader) error {
	if fileHeader == nil {
		return fmt.Errorf("file is required")
	}
	if fileHeader.Size <= 0 {
		return fmt.Errorf("file is empty")
	}
	if fileHeader.Size > MaxPDFSizeBytes {
		return fmt.Errorf("file is too large, max size is 20MB")
	}

	ext := strings.ToLower(filepath.Ext(fileHeader.Filename))
	if ext != ".pdf" {
		return fmt.Errorf("only .pdf files are allowed")
	}

	f, err := fileHeader.Open()
	if err != nil {
		return fmt.Errorf("failed to open file")
	}
	defer f.Close()

	head := make([]byte, 512)
	n, err := f.Read(head)
	if err != nil {
		return fmt.Errorf("failed to read file header")
	}
	contentType := http.DetectContentType(head[:n])
	if contentType != "application/pdf" {
		return fmt.Errorf("invalid file content type")
	}

	return nil
}
