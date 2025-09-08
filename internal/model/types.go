package model

type Chunk struct {
	ID   string `json:"id"`
	Text string `json:"text"`
}

type AskRequest struct {
	Query string `json:"query"`
	Model string `json:"model,omitempty"`
	TopK  int    `json:"topK,omitempty"`
}
