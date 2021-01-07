import React from "react";
import { GlyphCircle, GlyphCross, GlyphSquare } from "@vx/glyph";

import shapes from "../enums/shapes";

export default function BubbleCenter(props) {
  let Glyph;

  switch (props.shape) {
    case shapes.SQUARE:
      Glyph = GlyphSquare;
      break;
    case shapes.CROSS:
      Glyph = GlyphCross;
      break;
    default:
      Glyph = GlyphCircle;
  }

  return <Glyph left={props.x} top={props.y} size={50} fill={props.fill} />;
}
