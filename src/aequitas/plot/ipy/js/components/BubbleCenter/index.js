import React from "react";
import PropTypes from "prop-types";
import { GlyphCircle, GlyphCross, GlyphSquare } from "@vx/glyph";

import shapes from "~/constants/shapes";

const propTypes = {
  fill: PropTypes.string.isRequired,
  shape: PropTypes.string.isRequired,
  x: PropTypes.number.isRequired,
  y: PropTypes.number.isRequired,
};

function BubbleCenter(props) {
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

BubbleCenter.prototype = propTypes;
export default BubbleCenter;
