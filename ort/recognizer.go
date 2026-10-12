package ort

import (
	"github.com/oxzjh/cv"
	"github.com/oxzjh/lpr"
	"github.com/oxzjh/ort"
	"github.com/oxzjh/vector"
)

type RecognizeData struct {
	Tokens []int
}

type Recognizer struct {
	engine *ort.Ort
	width  int
	height int
	rows   int
	cols   int
}

func (r *Recognizer) Extract(aligned *cv.Mat) (tokens []int, score float32, err error) {
	resized, _ := aligned.ResizeWithBorder(r.width, r.height, cv.INTER_CUBIC, cv.BORDER_CONSTANT, cv.NewScalar(127.5, 127.5, 127.5, 0))
	inputBlob := resized.BlobFromImage(1/127.5, cv.Size{}, cv.NewScalar(127.5, 127.5, 127.5, 0), true, false)
	resized.Free()
	defer inputBlob.Free()
	var result *ort.Result
	if result, err = r.engine.Infer(inputBlob.Data(), inputBlob.Total()*inputBlob.ElemSize(), ort.TypeFloat); err != nil {
		return
	}
	defer result.Free()
	var data []float32
	if data, err = result.GetFloat32(0, r.rows*r.cols, false); err != nil {
		return
	}
	tokens, score = vector.FindTokens(data, r.rows, r.cols)
	return
}

func (r *Recognizer) Close() {
	if r.engine != nil {
		r.engine.Close()
		r.engine = nil
	}
}

func NewRecognizer(model string, threads int) (lpr.IRecognizer, error) {
	engine, err := ort.New(model, "recognizer", threads)
	if err != nil {
		return nil, err
	}
	return newRecognizer(engine), nil
}

func NewRecognizerWithBuffer(model []byte, threads int) (lpr.IRecognizer, error) {
	engine, err := ort.NewWithBuffer(model, "recognizer", threads)
	if err != nil {
		return nil, err
	}
	return newRecognizer(engine), nil
}

func newRecognizer(engine *ort.Ort) lpr.IRecognizer {
	return &Recognizer{engine, int(engine.InputDims[0][3]), int(engine.InputDims[0][2]), int(engine.OutputDims[0][1]), int(engine.OutputDims[0][2])}
}
