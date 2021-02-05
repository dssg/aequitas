import React from "react";
import PropTypes from "prop-types";

import Row from "./Row";
import sizes from "~/constants/sizes";

import "./style.scss";

const propTypes = {
  data: PropTypes.array.isRequired,
  handleActiveGroup: PropTypes.func.isRequired,
  metrics: PropTypes.array.isRequired,
  referenceGroup: PropTypes.string.isRequired,
  scaleBubbleSize: PropTypes.func.isRequired,
  scaleColor: PropTypes.func.isRequired,
  scalePosition: PropTypes.func.isRequired,
  scaleShape: PropTypes.func.isRequired,
  activeGroup: PropTypes.string,
  AxisComponent: PropTypes.object.isRequired,
  ThresholdsComponent: PropTypes.object,
  chartAreaHeight: PropTypes.number.isRequired,
  dataColumnNames: PropTypes.object.isRequired
};

function Chart(props) {
  const { AxisComponent, ThresholdsComponent } = props;
  return (
    <svg width={sizes.WIDTH} height={props.chartAreaHeight}>
      {AxisComponent}
      {ThresholdsComponent}
      {props.metrics.map((metric, index) => {
        const metricAxisY = sizes.MARGIN.top + sizes.ROW_HEIGHT * (index + 0.5);
        return (
          <Row
            key={`row-${metric}`}
            metric={metric}
            data={props.data}
            y={metricAxisY}
            referenceGroup={props.referenceGroup}
            scaleBubbleSize={props.scaleBubbleSize}
            scaleColor={props.scaleColor}
            scalePosition={props.scalePosition}
            scaleShape={props.scaleShape}
            activeGroup={props.activeGroup}
            handleActiveGroup={props.handleActiveGroup}
            dataColumnNames={props.dataColumnNames}
          />
        );
      })}
    </svg>
  );
}

Chart.propTypes = propTypes;

export default Chart;
