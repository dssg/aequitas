/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import PropTypes from "prop-types";
import React from "react";
import "./ErrorFallback.scss";

function ErrorFallback({ error }) {
    return (
        <div className="error" role="alert">
            <h3>Oops! Something went wrong!</h3>
            <p>Help us improve your experience by sending an error report</p>
            <pre>
                {error.name}: {error.message}
            </pre>
        </div>
    );
}

ErrorFallback.propTypes = {
    error: PropTypes.shape({
        message: PropTypes.string,
        name: PropTypes.string,
    }).isRequired,
};

export default ErrorFallback;
