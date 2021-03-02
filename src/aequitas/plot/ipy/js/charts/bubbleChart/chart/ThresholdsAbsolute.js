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

const SIDES = ["left", "right"];

function ThresholdsAbsolute(props) {
  function getSingleThresholdBand(side, metric, index) {
    const metricValue = find(
      props.data,
      (row) => row["attribute_value"] === props.referenceGroup
    )[`${metric.toLowerCase()}`];
    const y1 = sizes.AXIS.TOP.height + index * sizes.ROW_HEIGHT;
    let x, width, ruleX, thresholdValue;

    if (side === "left") {
      thresholdValue = metricValue / props.fairnessThreshold;
      // Left band starts with the innerChart:
      x = props.scalePosition.range()[0];
      // Position threshold rule on the thresholdValue:
      ruleX = props.scalePosition(thresholdValue);
      // Width of the left band should not be negative (when the threshold falls out of bounds),
      // and it should not exceed the width of the innerChart:
      width = Math.min(
        Math.max(ruleX - x, 0),
        props.scalePosition.range()[1] - props.scalePosition.range()[0]
      );
    } else if (side === "right") {
      thresholdValue = metricValue * props.fairnessThreshold;
      // Position threshold rule on the thresholdValue:
      ruleX = props.scalePosition(thresholdValue);
      // Right band starts on ruleX, if it falls within the innerChart boundaries:
      x = Math.min(
        Math.max(ruleX, props.scalePosition.range()[0]),
        props.scalePosition.range()[1]
      );
      // Width of the right band is the distance between its start (x)
      // and the end of the innerChart
      width = props.scalePosition.range()[1] - x;
    } else {
      return null;
    }

    const thresholdTooltipString = `${metric.toUpperCase()} = ${thresholdValue.toPrecision(
      2
    )}`;
    // From above, threshold rule can fall outside innerChart,
    // so we check if it should be displayed:
    const isRuleVisible =
      ruleX >= props.scalePosition.range()[0] &&
      ruleX <= props.scalePosition.range()[1];

    return (
      <ThresholdBand
        key={`${metric}-${side}`}
        thresholdColor={props.thresholdColor}
        fairnessThreshold={props.fairnessThreshold}
        svgKeySuffix={`${metric}-${side}`}
        x={x}
        ruleX={ruleX}
        y1={y1}
        y2={y1 + sizes.ROW_HEIGHT}
        width={width}
        thresholdTooltipString={thresholdTooltipString}
        displayRule={isRuleVisible}
      />
    );
  }

  return (
    <g>
      {props.metrics.map((metric, index) => {
        return SIDES.map((side) => getSingleThresholdBand(side, metric, index));
      })}
    </g>
  );
}

ThresholdsAbsolute.propTypes = propTypes;
export default ThresholdsAbsolute;
