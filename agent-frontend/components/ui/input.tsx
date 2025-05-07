import * as React from "react";

import { cn } from "@/lib/utils";
import { Button } from "./button";
import { Eye, EyeOff } from "lucide-react";

export type InputProps = React.ComponentProps<"input">;

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-base shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50 md:text-sm",
          className,
        )}
        ref={ref}
        {...props}
      />
    );
  },
);
Input.displayName = "Input";

interface PasswordInputProps extends Omit<InputProps, "type"> {
  /** Custom class name for the container */
  containerClassName?: string;
  /** Custom class name for the toggle button */
  toggleClassName?: string;
}

const PasswordInput = React.forwardRef<HTMLInputElement, PasswordInputProps>(
  ({ className, containerClassName, toggleClassName, ...props }, ref) => {
    const [showPassword, setShowPassword] = React.useState(false);

    return (
      <div className={cn("relative", containerClassName)}>
        <Input
          type={showPassword ? "text" : "password"}
          className={cn("pr-10", className)}
          ref={ref}
          {...props}
        />
        <Button
          variant="ghost"
          size="icon"
          onClick={(e) => {
            e.preventDefault();
            setShowPassword((prev) => !prev);
          }}
          className="absolute right-0 top-0 flex h-full items-center justify-center px-3 text-muted-foreground transition-colors hover:text-foreground"
        >
          {showPassword ? (
            <EyeOff className="size-3" />
          ) : (
            <Eye className="size-3" />
          )}
        </Button>
      </div>
    );
  },
);
PasswordInput.displayName = "PasswordInput";

export { Input, PasswordInput };
