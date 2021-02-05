import React from "react";
import PropTypes from "prop-types";
import { find } from "lodash";

import ThresholdBand from "~/components/ThresholdBand";
import sizes from "~/constants/sizes";
import "./style.scss";

const propTypes = {
  thresholdColor: PropTypes.string.isRequired,
  fairnessThreshold: PropTypes.number.isRequired,
  scalePosition: PropTypes.func.isRequired,
  metrics: PropTypes.array.isRequired,
  referenceGroup: PropTypes.string.isRequired,
  data: PropTypes.array.isRequired
};

function ThresholdsAbsolute(props) {
  function getSingleThresholdBand(side, metric, index) {
    let metricValue = find(
      props.data,
      (row) => row["attribute_value"] === props.referenceGroup
    )[`${metric.toLowerCase()}`];

    let x, width, ruleX, thresholdValue;

    if (side === "left") {
      thresholdValue = metricValue / props.fairnessThreshold;
      x = props.scalePosition.range()[0];
      ruleX = props.scalePosition(thresholdValue);
      width = ruleX - x;
    } else if (side === "right") {
      thresholdValue = Math.min(metricValue * props.fairnessThreshold, 1);
      x = props.scalePosition(thresholdValue);
      width = props.scalePosition.range()[1] - x;
      ruleX = x;
    } else {
      return null;
    }

    const thresholdDisplayString = `${metric.toUpperCase()} = ${thresholdValue.toPrecision(
      2
    )}`;

    return (
      <ThresholdBand
        key={`${side}-${metric}`}
        thresholdColor={props.thresholdColor}
        fairnessThreshold={props.fairnessThreshold}
        svgKeySuffix={`${side}-${metric}`}
        x={x}
        ruleX={ruleX}
        y1={sizes.MARGIN.top + index * sizes.ROW_HEIGHT}
        y2={sizes.MARGIN.top + (index + 1) * sizes.ROW_HEIGHT}
        width={width}
        thresholdDisplayString={thresholdDisplayString}
      />
    );
  }

  const sides = ["left", "right"];
  return (
    <g>
      {props.metrics.map((metric, index) => {
        return sides.map((side) => getSingleThresholdBand(side, metric, index));
      })}
    </g>
  );
}

ThresholdsAbsolute.propTypes = propTypes;
export default ThresholdsAbsolute;
