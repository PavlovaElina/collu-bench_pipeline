#lang racket
(define (change_base x base)
  (if (= x 0) "0"
      (let loop ([x x] [acc '()])
        (if (= x 0) (list->string acc)
            (loop (quotient x base)
                  (cons (integer->char (+ 48 (remainder x base))) acc))))))
