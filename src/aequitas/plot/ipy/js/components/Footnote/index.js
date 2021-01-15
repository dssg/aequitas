import React from "react";
import PropTypes from "prop-types";
import { format } from "d3-format";

import "./style.scss";
import colors from "~/constants/colors.scss";

function Footnote(props) {
  return (
    <div className="aequitas-footnote">
      <p
        style={{
          color: props.accessibilityMode
            ? colors.referenceGrey
            : colors.thresholdRed,
        }}
      >
        The metric value for any group should not be{" "}
        <span className="aequitas-bold-text">
          {format(".2")(Math.abs(props.fairnessThreshold))} (or more) times{" "}
        </span>
        smaller or larger than that of the reference group{" "}
        <span className="aequitas-bold-text">{props.referenceGroup}</span>
      </p>
    </div>
  );
}

Footnote.propTypes = {
  accessibilityMode: PropTypes.bool,
  fairnessThreshold: PropTypes.number,
  referenceGroup: PropTypes.string,
};

export default Footnote;
