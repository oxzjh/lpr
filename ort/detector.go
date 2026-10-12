package ort

import (
	"errors"

	"github.com/oxzjh/cv"
	"github.com/oxzjh/lpr"
	"github.com/oxzjh/ort"
)

type Detector struct {
	engine *ort.Ort
	width  int
	height int
	rows   int
	opts   *options
}

func (d *Detector) Detect(m *cv.Mat) (boxes []*cv.Box, err error) {
	resized, border := m.ResizeWithBorder(d.width, d.height, cv.INTER_CUBIC, cv.BORDER_CONSTANT, cv.Scalar{})
	inputBlob := resized.BlobFromImage(1/255.0, cv.Size{}, cv.Scalar{}, true, false)
	resized.Free()
	defer inputBlob.Free()
	var result *ort.Result
	if result, err = d.engine.Infer(inputBlob.Data(), inputBlob.Total()*inputBlob.ElemSize(), ort.TypeFloat); err != nil {
		return
	}
	defer result.Free()
	var detection []float32
	if detection, err = result.GetFloat32(0, d.rows*15, false); err != nil {
		return
	}
	for i := 0; i < d.rows; i++ {
		offset := i * 15
		if ss := detection[offset+4]; ss > d.opts.score {
			ox := detection[offset]
			oy := detection[offset+1]
			hw := detection[offset+2] * 0.5
			hh := detection[offset+3] * 0.5
			s0 := detection[offset+13]
			s1 := detection[offset+14]
			data := &lpr.DetectData{}
			box := &cv.Box{
				ID:   offset,
				X0:   ox - hw,
				Y0:   oy - hh,
				X1:   ox + hw,
				Y1:   oy + hh,
				Data: data,
			}
			if s0 >= s1 {
				box.Score = s0 * ss
				// data.Layer = 0
			} else {
				box.Score = s1 * ss
				data.Layer = 1
			}
			boxes = append(boxes, box)
		}
	}
	boxes = cv.NMS(boxes, d.opts.iou)
	if len(boxes) > 0 {
		ratio := 1 / border.Ratio
		for _, box := range boxes {
			box.X0 = (box.X0 - float32(border.Left)) * ratio
			box.Y0 = (box.Y0 - float32(border.Top)) * ratio
			box.X1 = (box.X1 - float32(border.Left)) * ratio
			box.Y1 = (box.Y1 - float32(border.Top)) * ratio

			data := box.Data.(*lpr.DetectData)
			for i := 0; i < 8; i += 2 {
				data.Landmarks[i] = (detection[box.ID+i+5] - float32(border.Left)) * ratio
				data.Landmarks[i+1] = (detection[box.ID+i+6] - float32(border.Top)) * ratio
			}
		}
	}

	return
}

func (d *Detector) DetectFile(file string) ([]*cv.Box, error) {
	m := cv.IMRead(file, cv.FLAG_COLOR)
	defer m.Free()
	if m.Cols() < 4 || m.Rows() < 4 {
		return nil, errors.New("read image failed")
	}
	return d.Detect(m)
}

func (d *Detector) Close() {
	if d.engine != nil {
		d.engine.Close()
		d.engine = nil
	}
}

func NewDetector(model string, threads int, opts ...Option) (lpr.IDetector, error) {
	engine, err := ort.New(model, "detector", threads)
	if err != nil {
		return nil, err
	}
	return newDetector(engine, opts)
}

func NewDetectorWithBuffer(model []byte, threads int, opts ...Option) (lpr.IDetector, error) {
	engine, err := ort.NewWithBuffer(model, "detector", threads)
	if err != nil {
		return nil, err
	}
	return newDetector(engine, opts)
}

func newDetector(engine *ort.Ort, opts []Option) (lpr.IDetector, error) {
	if engine.OutputDims[0][2] != 15 {
		return nil, errors.New("bad LPR model")
	}
	o := &options{score: 0.25, iou: 0.5}
	for _, opt := range opts {
		opt(o)
	}
	return &Detector{engine, int(engine.InputDims[0][3]), int(engine.InputDims[0][2]), int(engine.OutputDims[0][1]), o}, nil
}
