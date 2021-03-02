import React from "react";
import PropTypes from "prop-types";
import { AxisTop } from "@vx/axis";
import { GridColumns } from "@vx/grid";

import sizes from "~/constants/sizes";
import AxisInputs from "./AxisInputs";

import "./style.scss";

const getClassName = (isFaded) => {
  const boundClassName = "aequitas-absolute-boundary-ref-line";
  return isFaded ? boundClassName + "-faded" : boundClassName;
};
function AxisAbsolute(props) {
  const lowerBoundClassName = getClassName(props.axisBounds[0] !== 0);
  const upperBoundClassName = getClassName(props.axisBounds[1] !== 1);
  const showResetButton =
    props.axisBounds[0] !== 0 || props.axisBounds[1] !== 1;

  return (
    <g className="aequitas-axis">
      {showResetButton ? (
        <foreignObject
          x={0}
          y={sizes.AXIS.TOP.height / 2 - sizes.BOUNDS_INPUT.height / 2}
          /* Below, the * 1.2 is required so the inputs don't have their bounds cut */
          width={sizes.RESET_BUTTON.width * 1.2}
          height={sizes.BOUNDS_INPUT.height * 1.2}
        >
          <button
            style={{
              width: sizes.RESET_BUTTON.width,
              height: sizes.BOUNDS_INPUT.height
            }}
            className="reset-button"
            onClick={() => props.setAxisBounds([0, 1])}
          >
            Reset
          </button>
        </foreignObject>
      ) : null}
      <AxisTop
        scale={props.scale}
        top={sizes.AXIS.TOP.height}
        hideAxisLine
        hideTicks
      />
      <AxisInputs key={`axis-inputs-${props.axisBounds}`} {...props} />
      <GridColumns
        className="aequitas-grid"
        scale={props.scale}
        height={props.chartAreaHeight - sizes.AXIS.TOP.height}
        top={sizes.AXIS.TOP.height}
      />
      <line
        x1={sizes.AXIS.LEFT.width}
        x2={sizes.AXIS.LEFT.width}
        y1={sizes.AXIS.TOP.height}
        y2={props.chartAreaHeight}
        className={lowerBoundClassName}
      />
      <line
        x1={sizes.CHART_WIDTH - sizes.CHART_PADDING.right}
        x2={sizes.CHART_WIDTH - sizes.CHART_PADDING.right}
        y1={sizes.AXIS.TOP.height}
        y2={props.chartAreaHeight}
        className={upperBoundClassName}
      />
    </g>
  );
}

AxisAbsolute.propTypes = {
  chartAreaHeight: PropTypes.number,
  scale: PropTypes.func
};

export default AxisAbsolute;
