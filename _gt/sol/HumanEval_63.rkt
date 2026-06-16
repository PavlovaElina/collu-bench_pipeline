#lang racket
(define (fibfib n)
  (cond [(= n 0) 0] [(= n 1) 0] [(= n 2) 1]
        [else (let loop ([a 0] [b 0] [c 1] [i 3])
                (define d (+ a b c))
                (if (= i n) d (loop b c d (+ i 1))))]))
