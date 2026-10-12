package ort

type options struct {
	score float32
	iou   float32
}

type Option func(*options)

func WithScore(score float32) Option {
	return func(o *options) {
		o.score = score
	}
}

func WithIou(iou float32) Option {
	return func(o *options) {
		o.iou = iou
	}
}
