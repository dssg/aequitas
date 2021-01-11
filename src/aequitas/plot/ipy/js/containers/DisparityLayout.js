import React, { useState } from "react";
import PropTypes from "prop-types";
import { scaleOrdinal } from "d3-scale";
import { format } from "d3-format";

import DisparityChart from "./DisparityChart";
import Legend from "../components/Legend";

import { toTitleCase } from "../utils/helpers";

import colors from "../colors.scss";
import shapes from "../enums/shapes";

const propTypes = {
  accessibilityMode: PropTypes.bool.isRequired,
  attribute: PropTypes.string.isRequired,
  data: PropTypes.array.isRequired,
  fairnessThreshold: PropTypes.number.isRequired,
  referenceGroup: PropTypes.string.isRequired,
};

function DisparityLayout(props) {
  const [activeGroup, setActiveGroup] = useState(null);

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
          accessibilityMode={props.accessibilityMode}
          data={props.data}
          fairnessThreshold={props.fairnessThreshold}
          metrics={props.metrics}
          referenceGroup={props.referenceGroup}
          scaleColor={scaleColor}
          scaleShape={scaleShape}
          activeGroup={activeGroup}
          handleActiveGroup={setActiveGroup}
        />
        <p
          className="aequitas-annotation"
          style={{
            color: props.accessibilityMode
              ? colors.referenceGrey
              : colors.thresholdRed,
          }}
        >
          The metric value for any group should not be{" "}
          <span className="aequitas-bolder">
            {format(".2")(Math.abs(props.fairnessThreshold + 1))} (or more)
            times{" "}
          </span>
          smaller or larger than that of the reference group{" "}
          <span className="aequitas-bolder">{props.referenceGroup}</span>
        </p>
      </div>
      <Legend
        groups={groups}
        referenceGroup={props.referenceGroup}
        scaleColor={scaleColor}
        scaleShape={scaleShape}
        activeGroup={activeGroup}
        handleActiveGroup={setActiveGroup}
      />
    </div>
  );
}

DisparityLayout.propTypes = propTypes;
export default DisparityLayout;
