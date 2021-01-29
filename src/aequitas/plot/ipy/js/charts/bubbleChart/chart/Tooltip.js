import PropTypes from "prop-types";
import React from "react";
import { format } from "d3-format";

const propTypes = {
  data: PropTypes.object.isRequired,
  groupColor: PropTypes.string.isRequired,
  isReferenceGroup: PropTypes.bool.isRequired,
  metric: PropTypes.string.isRequired,
};

function Tooltip(props) {
  function renderRow(label, content) {
    return (
      <div className="aequitas-tooltip-item">
        <p className="aequitas-tooltip-item-label">{label}:</p>
        <p>{content}</p>
      </div>
    );
  }

  function renderDisparityValueExplainerText(value, metric) {
    const comparatorWord = value > 0 ? "larger" : "smaller";
    const formattedValue = format(".2")(Math.abs(value) + 1);
    return (
      <span>
        {`${formattedValue} times ${comparatorWord} ${metric.toUpperCase()} `}
        <span className="aequitas-text-explainer">{` than the reference group`}</span>
      </span>
    );
  }

  function renderGroupSize(groupSize, totalEntities) {
    const formattedGroupSize = format(".2~s")(groupSize).replace(/G/, "B");
    return (
      <span>
        {formattedGroupSize}
        <span className="aequitas-text-explainer">{` (${format(".2%")(
          groupSize / totalEntities
        )})`}</span>
      </span>
    );
  }
  return (
    <div
      className="aequitas-tooltip"
      style={{ borderLeftColor: props.groupColor }}
    >
      {renderRow("Group", props.data["attribute_value"])}
      {renderRow(
        "Group Size",
        renderGroupSize(props.data["group_size"], props.data["total_entities"])
      )}
      {renderRow(
        "Disparity",
        props.isReferenceGroup
          ? "Reference Group"
          : renderDisparityValueExplainerText(
              props.data[`${props.metric}_disparity_scaled`],
              props.metric
            )
      )}
      {renderRow(
        props.metric.toUpperCase(),
        format(".2")(props.data[props.metric])
      )}
    </div>
  );
}

Tooltip.propTypes = propTypes;
export default Tooltip;
