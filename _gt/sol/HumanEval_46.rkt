#lang racket
(define (fib4 n)
  (cond [(= n 0) 0] [(= n 1) 0] [(= n 2) 2] [(= n 3) 0]
        [else (let loop ([a 0] [b 0] [c 2] [d 0] [i 4])
                (define e (+ a b c d))
                (if (= i n) e (loop b c d e (+ i 1))))]))
