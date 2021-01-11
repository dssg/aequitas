import React from "react";
import PropTypes from "prop-types";
import { range } from "lodash";
import { format } from "d3-format";
import { AxisTop } from "@vx/axis";
import { Text } from "@vx/text";
import { GridColumns } from "@vx/grid";

import sizesEnum from "../enums/sizes";

const propTypes = {
  chartAreaHeight: PropTypes.number.isRequired,
  scale: PropTypes.func.isRequired,
};

function getTickValues(limit) {
  const TICK_STEP_OPTIONS = [1, 2, 5, 10, 20, 50, 100];
  let tickValues = [];

  for (let tickStep of TICK_STEP_OPTIONS) {
    if (limit / tickStep <= 6 || tickStep === [...TICK_STEP_OPTIONS].pop()) {
      const tickStart = Math.ceil(limit / tickStep) * tickStep;
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
}

function DisparityAxis(props) {
  const tickValues = getTickValues(props.scale.domain()[1]);
  function renderTextAnnotations() {
    return (
      <g>
        <Text
          x={sizesEnum.MARGIN.left}
          y={0}
          textAnchor="start"
          verticalAnchor="start"
        >
          Times Smaller
        </Text>
        <Text
          x={sizesEnum.WIDTH - sizesEnum.MARGIN.right}
          y={0}
          textAnchor="end"
          verticalAnchor="start"
        >
          Times Larger
        </Text>
        <Text
          x={props.scale(0)}
          y={0}
          textAnchor="middle"
          verticalAnchor="start"
        >
          Equal
        </Text>
      </g>
    );
  }
  return (
    <g className="aequitas-axis">
      {renderTextAnnotations()}
      <AxisTop
        scale={props.scale}
        top={sizesEnum.MARGIN.top}
        hideAxisLine
        hideTicks
        tickFormat={(value) =>
          value === 0 ? "=" : format("d")(Math.abs(value) + 1)
        }
        tickValues={tickValues}
      />
      <GridColumns
        scale={props.scale}
        height={props.chartAreaHeight}
        top={sizesEnum.MARGIN.top}
        className="aequitas-grid"
        tickValues={tickValues}
      />
      <line
        x1={props.scale(0)}
        x2={props.scale(0)}
        y1={sizesEnum.MARGIN.top}
        y2={sizesEnum.MARGIN.top + props.chartAreaHeight}
        className="aequitas-equal-ref-line"
      />
    </g>
  );
}

DisparityAxis.propTypes = propTypes;
export default DisparityAxis;
