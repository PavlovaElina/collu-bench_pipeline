#lang racket
(define (fib n)
  (let loop ([a 0] [b 1] [i 0])
    (if (= i n) a (loop b (+ a b) (+ i 1)))))
