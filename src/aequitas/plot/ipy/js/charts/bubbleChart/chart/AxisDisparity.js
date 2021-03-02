import React from "react";
import PropTypes from "prop-types";
import { range } from "lodash";
import { format } from "d3-format";
import { AxisTop } from "@vx/axis";
import { Text } from "@vx/text";
import { GridColumns } from "@vx/grid";

import sizes from "~/constants/sizes";

const getTickValues = (maxTickValue) => {
  const TICK_STEP_OPTIONS = [1, 2, 5, 10, 20, 50, 100];
  let tickValues = [];

  for (let tickStep of TICK_STEP_OPTIONS) {
    if (
      maxTickValue / tickStep <= 6 ||
      tickStep === [...TICK_STEP_OPTIONS].pop()
    ) {
      const tickStart = Math.ceil(maxTickValue / tickStep) * tickStep;
      tickValues = range(-tickStart, tickStart + 1, tickStep);
      if (tickStep > 1) {
        tickValues = tickValues.map((value) => {
          if (value === 0) {
            return value;
          }

          if (value > 0) {
            return value - 1;
          }

          return value + 1;
        });
      }
      break;
    }
  }
  return tickValues;
};

function AxisDisparity(props) {
  const tickValues = getTickValues(props.scale.domain()[1]);

  function renderTextAnnotations(text, x, anchor) {
    return (
      <Text x={x} y={0} textAnchor={anchor} verticalAnchor="start">
        {text}
      </Text>
    );
  }

  return (
    <g className="aequitas-axis">
      {renderTextAnnotations("Times Smaller", props.scale.range()[0], "start")}
      {renderTextAnnotations("Equal", props.scale(0), "middle")}
      {renderTextAnnotations("Times Larger", props.scale.range()[1], "end")}

      <AxisTop
        scale={props.scale}
        top={sizes.AXIS.TOP.height}
        tickFormat={(value) =>
          value === 0 ? "=" : format("d")(Math.abs(value) + 1)
        }
        tickValues={tickValues}
        hideAxisLine
        hideTicks
      />
      <GridColumns
        className="aequitas-grid"
        scale={props.scale}
        height={props.chartAreaHeight}
        top={sizes.AXIS.TOP.height}
        tickValues={tickValues}
      />
      <line
        x1={props.scale(0)}
        x2={props.scale(0)}
        y1={sizes.AXIS.TOP.height}
        y2={sizes.AXIS.TOP.height + props.chartAreaHeight}
        className="aequitas-equal-ref-line"
      />
    </g>
  );
}

AxisDisparity.propTypes = {
  chartAreaHeight: PropTypes.number,
  scale: PropTypes.func
};

export default AxisDisparity;
