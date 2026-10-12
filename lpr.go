package lpr

import (
	"errors"

	"github.com/oxzjh/cv"
)

type LPR struct {
	detector   *Detector
	recognizer *Recognizer
}

func (l *LPR) Detect(m *cv.Mat) ([]*License, error) {
	boxes, err := l.detector.Detect(m)
	if err != nil {
		return nil, err
	}
	licenses := make([]*License, 0, len(boxes))
	for _, box := range boxes {
		data := box.Data.(*DetectData)
		aligned, width, height := m.AlignCropRect(data.Landmarks)
		defer aligned.Free()
		var (
			tokens []int
			score  float32
		)
		if data.Layer == 1 { // double
			y := height * 0.4
			up := aligned.Region(cv.NewRect(0, 0, int(width), int(y)))
			upTokens, upScore, err := l.recognizer.Extract(up)
			up.Free()
			if err != nil {
				return nil, err
			}
			down := aligned.Region(cv.NewRect(0, int(y), int(width), int(height-y)))
			downTokens, downScore, err := l.recognizer.Extract(down)
			down.Free()
			if err != nil {
				return nil, err
			}
			tokens = append(upTokens, downTokens...)
			score = (upScore + downScore) * 0.5
		} else {
			if tokens, score, err = l.recognizer.Extract(aligned); err != nil {
				return nil, err
			}
		}
		if len(tokens) >= 7 {
			licenses = append(licenses, NewLicense(box, data, tokens, score))
		}
	}
	return licenses, nil
}

func (l *LPR) DetectFile(file string) ([]*License, error) {
	m := cv.IMRead(file, cv.FLAG_COLOR)
	if m.Rows() < 4 || m.Cols() < 4 {
		return nil, errors.New("read file failed")
	}
	defer m.Free()
	return l.Detect(m)
}

func (l *LPR) Close() {
	l.detector.Close()
	l.recognizer.Close()
}

func NewDefault(detectorModel, recognizerModel string, threads int) (*LPR, error) {
	return New(
		detectorModel,
		recognizerModel,
		0.25, 0.5, threads,
	)
}

func New(detectorModel, recognizerModel string, score, iou float32, threads int) (*LPR, error) {
	detector, err := NewDetector(detectorModel, threads, score, iou)
	if err != nil {
		return nil, err
	}

	recognizer, err := NewRecognizer(recognizerModel, threads)
	if err != nil {
		detector.Close()
		return nil, err
	}

	return &LPR{detector, recognizer}, nil
}
