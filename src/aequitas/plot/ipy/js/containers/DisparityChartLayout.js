import React, { useState } from "react";
import { scaleOrdinal } from "d3-scale";
import DisparityChart from "./DisparityChart";
import Legend from "../components/Legend";

import { toTitleCase } from "../utils/helpers";

import colors from "../colors.scss";
import shapes from "../enums/shapes";

export default function DisparityChartLayout(props) {
  const [hoverGroup, setHoverGroup] = useState(null);

  const groups = props.data.map((row) => row["attribute_value"]);
  const sortedGroups = [
    props.referenceGroup,
    ...groups.filter((item) => item !== props.referenceGroup),
  ];

  const scaleColor = scaleOrdinal()
    .domain(sortedGroups)
    .range([colors.referenceGrey].concat(colors.categoricalPalette.split(",")));

  let shapeRange = [shapes.CROSS].concat(
    Array(groups.length - 1).fill(shapes.CIRCLE)
  );

  if (props.accessibilityMode) {
    shapeRange = [shapes.CROSS]
      .concat(Array(Math.ceil((groups.length - 1) / 2)).fill(shapes.CIRCLE))
      .concat(Array(Math.floor((groups.length - 1) / 2)).fill(shapes.SQUARE));
  }

  const scaleShape = scaleOrdinal().domain(sortedGroups).range(shapeRange);

  return (
    <div className="aequitas-chart-area">
      <div className="aequitas-chart">
        <h1>Disparities on {toTitleCase(props.attribute)}</h1>
        <DisparityChart
          {...props}
          scaleColor={scaleColor}
          scaleShape={scaleShape}
          hoverGroup={hoverGroup}
          handleHoverGroup={setHoverGroup}
        />
      </div>
      <Legend
        groups={groups}
        referenceGroup={props.referenceGroup}
        scaleColor={scaleColor}
        scaleShape={scaleShape}
        hoverGroup={hoverGroup}
        handleHoverGroup={setHoverGroup}
      />
    </div>
  );
}
