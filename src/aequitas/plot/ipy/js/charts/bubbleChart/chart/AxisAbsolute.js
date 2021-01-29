import React from "react";
import PropTypes from "prop-types";
import { AxisTop } from "@vx/axis";
import { GridColumns } from "@vx/grid";

import sizes from "~/constants/sizes";

function AxisAbsolute(props) {
  return (
    <g className="aequitas-axis">
      <AxisTop
        scale={props.scale}
        top={sizes.MARGIN.top}
        // tickFormat=".1f"
        // tickValues={tickValues}
        hideAxisLine
        hideTicks
      />
      <GridColumns
        className="aequitas-grid"
        scale={props.scale}
        height={props.chartAreaHeight}
        top={sizes.MARGIN.top}
        // tickValues={tickValues}
      />
      <line
        x1={props.scale(0)}
        x2={props.scale(0)}
        y1={sizes.MARGIN.top}
        y2={sizes.MARGIN.top + props.chartAreaHeight}
        className="aequitas-absolute-boundary-ref-line"
      />
      <line
        x1={props.scale(1)}
        x2={props.scale(1)}
        y1={sizes.MARGIN.top}
        y2={sizes.MARGIN.top + props.chartAreaHeight}
        className="aequitas-absolute-boundary-ref-line"
      />
    </g>
  );
}

AxisAbsolute.propTypes = {
  chartAreaHeight: PropTypes.number,
  scale: PropTypes.func
};

export default AxisAbsolute;
