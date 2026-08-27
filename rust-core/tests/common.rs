// common wrapper for test file

macro_rules! test_me {
    (
        $(#[$attr:meta])*
        $name: ident,
        $func: expr
        $(, $arg: expr)*
    ) => {
        $(#[$attr])*
        #[test]
        fn $name() {
            $func($($arg),*);
        }
    };

    // passign batch of test to it
    (
        $func: expr;
        $(
            $(#[$attr:meta])*
            $name: ident,
            $(, $arg: expr)*
        ), * $(,)?
    ) => {
        $test_me!($(#[$attr])* $name, $func, $(, $arg)*);
    }
}
