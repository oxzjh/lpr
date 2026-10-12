package lpr

import (
	"image"

	"github.com/oxzjh/cv"
)

var gRunes = []rune{' ', '\'', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '云', '京', '冀', '吉', '学', '宁', '川', '挂', '新', '晋', '桂', '民', '沪', '津', '浙', '渝', '港', '湘', '琼', '甘', '皖', '粤', '航', '苏', '蒙', '藏', '警', '豫', '贵', '赣', '辽', '鄂', '闽', '陕', '青', '鲁', '黑', '领', '使', '澳'}

type License struct {
	Rect  image.Rectangle
	Layer int
	Shape [4]image.Point
	Value string
	Score float32
}

func NewLicense(box *cv.Box, data *DetectData, tokens []int, score float32) *License {
	result := &License{
		Rect: image.Rect(
			int(box.X0+0.5),
			int(box.Y0+0.5),
			int(box.X1+0.5),
			int(box.Y1+0.5),
		),
		Layer: data.Layer,
		Score: score,
	}

	for i := range result.Shape {
		offset := i << 1
		result.Shape[i].X = int(data.Landmarks[offset] + 0.5)
		result.Shape[i].Y = int(data.Landmarks[offset+1] + 0.5)
	}

	runes := make([]rune, len(tokens))
	for i, v := range tokens {
		runes[i] = gRunes[v]
	}
	result.Value = string(runes)

	return result
}
