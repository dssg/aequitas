import React from "react";
import { format } from "d3-format";
import { AxisTop } from "@vx/axis";
import { Text } from "@vx/text";
import { GridColumns } from "@vx/grid";

import sizesEnum from "../enums/sizes";

export default function DisparityAxis(props) {
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
        x={
          (sizesEnum.MARGIN.left + sizesEnum.WIDTH - sizesEnum.MARGIN.right) / 2
        }
        y={0}
        textAnchor="middle"
        verticalAnchor="start"
      >
        Equal
      </Text>
      <AxisTop
        scale={props.scale}
        top={sizesEnum.MARGIN.top}
        hideAxisLine
        hideTicks
        tickFormat={(value) => format("d")(Math.abs(value))}
      />
      <GridColumns
        scale={props.scale}
        height={props.chartAreaHeight}
        top={sizesEnum.MARGIN.top}
        className="aequitas-grid"
      />
    </g>
  );
}
