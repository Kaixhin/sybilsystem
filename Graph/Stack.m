classdef Stack < handle
  %STACK Implementation of the stack data type
  properties (Hidden)
    internal
  end
  methods (Static)
    % Constructor
    function this = Stack()
      this.internal = {};
    end
  end
  methods
    % Push element on top of stack
    function push(this, el)
      this.internal = [el this.internal];
    end
    % Pop element from top of stack
    function el = pop(this)
      el = [];
      if (size(this.internal, 2) > 0)
        el = this.internal{1};
        this.internal(1) = [];
      end
    end
    % Peek at element at top of stack
    function el = peek(this)
      el = [];
      if (size(this.internal, 2) > 0)
        el = this.internal{1};
      end
    end
  end  
end