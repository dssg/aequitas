import React from "react";
import { format } from "d3-format";
import { AxisTop } from "@vx/axis";
import { Text } from "@vx/text";
import { GridColumns } from "@vx/grid";

import sizesEnum from "../enums/sizes";

export default function DisparityAxis(props) {
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
    <g>
      {renderTextAnnotations()}
      <AxisTop
        scale={props.scale}
        top={sizesEnum.MARGIN.top}
        hideAxisLine
        hideTicks
        tickFormat={(value) =>
          value === 0 ? "=" : format("d")(Math.abs(value) + 1)
        }
        // todo tickValues logic to override 31, 41, 51...
      />
      <GridColumns
        scale={props.scale}
        height={props.chartAreaHeight}
        top={sizesEnum.MARGIN.top}
        className="aequitas-grid"
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
