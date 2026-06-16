#lang racket
(define (encrypt s)
  (list->string
   (map (lambda (c)
          (cond [(char<=? #\a c #\z)
                 (integer->char (+ 97 (modulo (+ (- (char->integer c) 97) 4) 26)))]
                [(char<=? #\A c #\Z)
                 (integer->char (+ 65 (modulo (+ (- (char->integer c) 65) 4) 26)))]
                [else c]))
        (string->list s))))
