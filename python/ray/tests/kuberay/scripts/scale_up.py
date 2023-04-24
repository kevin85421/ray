import textwrap
import logging

import ray
from ray._private import test_utils


@test_utils.wait_for_stdout(
    strings_to_match=["Adding 1 node(s) of type small-group."], timeout_s=15
)
def main():
    """Submits CPU request.
    Wait 15 sec for autoscaler scale-up event to get emitted to stdout.

    The autoscaler update interval is 5 sec, so it should be enough to wait 5 seconds.
    An extra ten seconds are added to the timeout as a generous buffer against
    flakiness.
    """
    loggers = {f"{logging.root}": logging.root}
    loggers.update(dict(sorted(logging.root.manager.loggerDict.items())))
    msg = []
    for n, l in sorted(loggers.items()):
        msg.append(f"{n}")
        if isinstance(l, logging.Logger):
            for h in l.handlers:
                msg.append(textwrap.indent(str(h), "  "))

    logger = logging.getLogger(__name__)
    logger.critical("~~~~~~~~~~~~~~~scale_up.py~~~~~~~~~~~~~~")
    logger.critical("\n".join(msg))
    ray.autoscaler.sdk.request_resources(num_cpus=2)


if __name__ == "__main__":
    ray.init("auto")
    main()
